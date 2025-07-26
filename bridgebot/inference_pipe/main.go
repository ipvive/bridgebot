package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net"
	"net/http"
	_ "net/http/pprof"
	"sync"
	"time"
	"os"
	"os/signal"
	"syscall"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/golang/protobuf/proto"

	"ipvive/bridgebot/pb"
)

var (
	certFile   = flag.String("cert_file", "", "The TLS cert file")
	keyFile    = flag.String("key_file", "", "The TLS key file")
	listen     = flag.String("listen", "localhost:20000", "The server listen address")
	listenInfo = flag.String("info", "", "port to provide monitoring and profiling information")
	maxPending = flag.Int("max_pending", 4096, "maximum number of pending requests")
	maxDelay   = flag.Int("max_delay_ms", 0, "maximum delay for waiting for batch in ms")
)

type req struct {
	f *pb.FeaturesMicroBatch
	c chan *pb.PredictionsMicroBatch
}

type inflight struct {
	batch []*req
	exp time.Time
}

type metrics struct {
	inferencesProcessed prometheus.Counter
}

type server struct {
	pending chan *req
	mu sync.Mutex
	inflight map[uint64]*inflight
	m metrics
	next uint64
}

func NewServer(sz int) *server {
	m := metrics{
		inferencesProcessed: promauto.NewCounter(prometheus.CounterOpts{
			Name: "inference_pipe_inferences_processed"}),
	}
	return &server{
		pending: make(chan *req, sz),
		inflight: make(map[uint64]*inflight),
		m: m,
	}
}

func (s *server) Predict(ctx context.Context, in *pb.FeaturesMicroBatch) (*pb.PredictionsMicroBatch, error) {
	c := make(chan *pb.PredictionsMicroBatch)
	r := req{f:proto.Clone(in).(*pb.FeaturesMicroBatch), c:c}
	s.pending <- &r
	resp := <-r.c
	return resp, nil
}

func (s *server) GetFeaturesBatch(ctx context.Context, in *pb.FeaturesBatchRequest) (*pb.FeaturesBatch, error) {
	//log.Printf("Get Features Batch (%d/%d)", len(s.pending), in.BatchSize)
	s.expire()
	sz := int(in.BatchSize)
	batch := make([]*req, sz)
	var timeout <-chan time.Time
	if *maxDelay > 0 {
		timeout = time.After(time.Duration(*maxDelay) * time.Millisecond)
	}
ForLoop:
	for i := 0; i < sz; i++ {
		select {
		case batch[i] = <-s.pending:
		case <-timeout:
			log.Printf("Sending partial features batch (%d/%d)", i, in.BatchSize)
			batch = batch[:i]
			break ForLoop
		}
	}
	i := &inflight{batch:batch, exp:time.Now().Add(time.Minute)}
	s.mu.Lock()
		id := s.next
		s.next = s.next + 1
		s.inflight[id] = i
	s.mu.Unlock()

	micro := make([]*pb.FeaturesMicroBatch, sz)
	for i, f := range batch {
		micro[i] = f.f
	}
	return &pb.FeaturesBatch{UniqueId:id, Micro:micro}, nil
}

func (s *server) PutPredictionsBatch(ctx context.Context, in *pb.PredictionsBatch) (*pb.Empty, error) {
	//log.Printf("Put Predictions Batch (%d)", len(s.inflight))
	s.mu.Lock()
		i, ok := s.inflight[in.UniqueId]
		if ok {
			delete(s.inflight, in.UniqueId)
		}
	s.mu.Unlock()

	if !ok {
		log.Printf("Id %v not found", in.UniqueId)
		return nil, errors.New("Id not found")
	}

	if len(i.batch) != len(in.Micro) {
		log.Printf("request/response size mismatch: %d != %d",
			len(i.batch), len(in.Micro))
	} else {
		for j, req := range i.batch {
			req.c <- proto.Clone(in.Micro[j]).(*pb.PredictionsMicroBatch)
			s.m.inferencesProcessed.Add(float64(len(in.Micro[j].Prediction)))
		}
	}

	return &pb.Empty{}, nil
}

func (s *server) expire() {
	t := time.Now()
	var redo []*req
	s.mu.Lock()
	for uid, i := range s.inflight {
		if i.exp.Before(t) {
			redo = append(redo, i.batch...)
			delete(s.inflight, uid)
		}
	}
	s.mu.Unlock()
	if len(redo) > 0 {
		log.Printf("Expired %d inferences (%d inflight remain)", len(redo), len(s.inflight))
	}
	for i := range redo {
		s.pending <- redo[i]
	}
}

func (s *server) rewind() {
	s.mu.Lock()
	redo := s.inflight
	s.inflight = make(map[uint64]*inflight)
	s.mu.Unlock()

	for _, i := range redo {
		for _, r := range i.batch {
			s.pending <- r
		}
	}
}

func main() {
	flag.Parse()
	if *listenInfo != "" {
		http.Handle("/metrics", promhttp.Handler())
		go func() { http.ListenAndServe(*listenInfo, nil) }()
	}
	s := NewServer(*maxPending)
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
	pb.RegisterInferencePipeServer(grpcServer, s)
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGHUP)
	go func() { for { _ = <-c; log.Printf("Rewinding."); s.rewind() } }()
	grpcServer.Serve(lis)
	log.Printf("Exiting.")
}
