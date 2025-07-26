package main

import (
	"context"
	"flag"
	"log"

	"google.golang.org/grpc"

	"ipvive/bridgebot/pb"
)

var (
	pipeAddr     = flag.String("pipe_addr", "localhost:20000", "The inference pipe address")
	numClients   = flag.Int("num_clients", 10, "Number of simultaneous requests")
)

func main() {
	flag.Parse()
	conn, err := grpc.Dial(*pipeAddr, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("failed to dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewInferencePipeClient(conn)
	req := &pb.FeaturesMicroBatch{}
	ctx := context.Background()
	for i := 0; i <= *numClients; i++ {
		go func() { for { _, err := client.Predict(ctx, req); if err != nil { log.Fatal(err) } } }()
	}
	c := make(chan struct{})
	_ = <-c
}
