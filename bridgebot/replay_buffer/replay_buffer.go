package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	_ "net/http/pprof"
	"sync"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"golang.org/x/exp/rand"
	"github.com/golang/protobuf/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/reflection"
	"gonum.org/v1/gonum/stat/sampleuv"
	"github.com/ryszard/tfutils/go/tfrecord"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/io/filesystem"
	_ "github.com/apache/beam/sdks/v2/go/pkg/beam/io/filesystem/gcs"
	_ "github.com/apache/beam/sdks/v2/go/pkg/beam/io/filesystem/local"

	"ipvive/bridgebot/pb"
)

var (
	certFile   = flag.String("cert_file", "", "The TLS cert file")
	keyFile    = flag.String("key_file", "", "The TLS key file")
	listen     = flag.String("listen", "localhost:10000", "The server listen address")
	serveBoards = flag.String("serve_boards", "", "comma-separated-list of file globs of boards to read and serve.")
	newBoards = flag.String("new_boards", "./data/replay/boards-NNNNN.tfrec", "Filepath tempate for new boards.")
	boardsPerFile = flag.Int("boards_per_file", 1000, "maximum number of boards per new file created")
	bufferSize = flag.Int("buffer_size", 2000, "maximum number of boards to keep in buffer")
)

type memBoard struct {
	n int
	b []byte
}

type server struct {
	mu sync.Mutex
	boards []memBoard
	w chan []byte
}

func (s *server) Put(ctx context.Context, in *pb.PlayedBoard) (*pb.PutResponse, error) {
	var err error

	b, err := proto.Marshal(in)
	if err != nil {
		return nil, err
	}
	n := num_positions(in)

	s.mu.Lock()
	s.boards = append(s.boards, memBoard{n:n, b:b})
	diff := len(s.boards) - *bufferSize
	if diff > 0 {
		s.boards = s.boards[diff:]
	}
	w := s.w
	s.mu.Unlock()

	w <- b
	return &pb.PutResponse{}, nil
}

func num_positions(in *pb.PlayedBoard) int {
	n := 0
	for _, g := range in.Tables {
		n += len(g.Actions)
	}
	return n
}

func position_n(in *pb.PlayedBoard, n int) *pb.PlayedBoardPosition {
	ti, ai := 0, n
	for ai >= len(in.Tables[ti].Actions) {
		ti, ai = ti + 1, ai - len(in.Tables[ti].Actions)
	}
	return &pb.PlayedBoardPosition{Board:in, TableIndex:uint32(ti), ActionIndex:uint32(ai)}
}

func (s *server) SampleBatch(ctx context.Context, in *pb.SampleBatchRequest) (*pb.PlayedBoardBatch, error) {

	// initialize weights
	s.mu.Lock()
	boards := s.boards
	s.mu.Unlock()

	log.Print("Sampling %d positions", in.BatchSize)
	if len(boards) < int(in.BatchSize) {
		return nil, errors.New("Not enough boards for sample.")
	}
	weights := make([]float64, len(boards))
	for i, v := range boards {
		weights[i] = float64(v.n)
	}
	positions := make([]*pb.PlayedBoardPosition, in.BatchSize)
	sampler := sampleuv.NewWeighted(weights, nil)
	for i := 0; i < int(in.BatchSize); i++ {
		j, ok := sampler.Take()
		if !ok {
			panic("sample failed")
		}
		board := &pb.PlayedBoard{}
		if err := proto.Unmarshal(boards[j].b, board); err != nil {
			panic("unmarshal failed")
		}
		n := rand.Intn(boards[j].n)
		positions[i] = position_n(board, n)
	}
	return &pb.PlayedBoardBatch{Position: positions}, nil
}

func (s *server) Read(ctx context.Context, fileglobs []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, g := range fileglobs {
		fs, err := filesystem.New(ctx, g)
		if err != nil {
			log.Fatal(err)
		}
		matches, err := fs.List(ctx, g)
		if err != nil {
			log.Fatal(err)
		}
		for _, m := range matches {
			f, err := fs.OpenRead(ctx, m)
			if err != nil {
				log.Fatal(err)
			}
			i := 0
			for ; ; i++ {
				b, err := tfrecord.Read(f)
				if err == io.EOF {
					break
				} else if err != nil {
					log.Printf("While reading record %v from %v: %v", i, m, err)
					break
				}
				board := &pb.PlayedBoard{}
				if err := proto.Unmarshal(b, board); err != nil {
					log.Printf("malformed board: %v", err)
					break
				}
				n := num_positions(board)
				if n > 0 {
					s.boards = append(s.boards, memBoard{n:n, b:b})
					diff := len(s.boards) - *bufferSize
					if diff > 0 {
						s.boards = s.boards[diff:]
					}
				}
			}
			log.Printf("Read %v records from %v", i, m)
			f.Close()
		}
	}
}

func filenameFromTemplate(t string, n int) (string, error) {
	pos := strings.LastIndex(t, "N")
	if pos < 0 {
		return "", errors.New("missing N in template")
	}
	var i int
	for i = pos; i >= 0 && t[i] == 'N'; i-- {}
	ns := fmt.Sprintf("%0*d", pos - i, n)
	if len(ns) > pos - i {
		return "", errors.New("decimal overflow")
	}
	return t[:i + 1] + ns + t[pos + 1:], nil
}

func globFromTemplate(t string) (string, error) {
	pos := strings.LastIndex(t, "N")
	if pos < 0 {
		return "", errors.New("missing N in template")
	}
	var i int
	for i = pos; i >= 0 && t[i] == 'N'; i-- {}
	return t[:i + 1] + "*" + t[pos + 1:], nil
}

func (s *server) Write(ctx context.Context, filetemplate string, num_per_file int) {
	s.mu.Lock()
	if s.w == nil {
		s.w = make(chan []byte, 100)
	}
	ch := s.w
	s.mu.Unlock()
	fs, err := filesystem.New(ctx, filetemplate)
	if err != nil {
		log.Fatal(err)
	}
	pat, err := globFromTemplate(filetemplate)
	if err != nil {
		log.Fatal(err)
	}
	existing, err := fs.List(ctx, pat)
	if err != nil {
		log.Fatal(err)
	}
	for i := 0; ; i++ {
		name, err := filenameFromTemplate(filetemplate, i)
		if err != nil {
			log.Fatal(err)
		}
		nameExists := false
		for _, e := range existing {
			if name == e {
				nameExists = true
			}
		}
		if nameExists {
			continue
		}
		f, err := fs.OpenWrite(ctx, name)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Writing new boards to %v", name)
		for j := 0; j < num_per_file; j++ {
			b, more := <-ch
			if more {
				if err := tfrecord.Write(f, b); err != nil {
					log.Fatal(err)
				}
				log.Print("Record written.")
			} else {
				f.Close()
				log.Print("Write finished.")
				return
			}
		}
		f.Close()
		log.Printf("Wrote %v records to %v", num_per_file, name)
	}
}

func (s *server) Close() error {
	s.mu.Lock()
	close(s.w)
	s.mu.Unlock()
	return nil
}

func main() {
	flag.Parse()
	go func() { http.ListenAndServe("0.0.0.0:3000", nil) } ()
	ctx := context.Background()
	s := &server{}
	s.Read(ctx, strings.Split(*serveBoards, ","))

	lis, err := net.Listen("tcp", *listen)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	var opts []grpc.ServerOption
	if *certFile != "" || *keyFile != "" {
		creds, err := credentials.NewServerTLSFromFile(*certFile, *keyFile)
		if err != nil {
			log.Fatalf("Failed to generate credentials %v", err)
		}
		opts = []grpc.ServerOption{grpc.Creds(creds)}
	}
	grpcServer := grpc.NewServer(opts...)
	pb.RegisterReplayBufferServer(grpcServer, s)
	reflection.Register(grpcServer)
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt, syscall.SIGHUP)
	log.Printf("%d boards loaded.", len(s.boards))
	log.Printf("Serving on %v.", *listen)
	go grpcServer.Serve(lis)
	go func() { _ = <-quit; log.Printf("Shutting down."); grpcServer.Stop(); s.Close()}()
	s.Write(ctx, *newBoards, *boardsPerFile)
}

