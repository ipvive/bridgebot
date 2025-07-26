package main

import (
	"context"
	"flag"
	"log"
	"time"

	"google.golang.org/grpc"

	"ipvive/bridgebot/pb"
)

var (
	pipeAddr     = flag.String("pipe_addr", "localhost:20000", "The inference pipe address")
	numClients   = flag.Int("num_clients", 1, "Number of simultaneous requests")
	batchSize   = flag.Int("batch_size", 100, "batch size for requests")
	latency     = flag.String("latency", "0.1s", "time between request and response")
)

const (
	numBridgeActions = 90
)

func main() {
	flag.Parse()
	sleepInterval, err := time.ParseDuration(*latency)
	if err != nil {
		log.Fatalf("failed to parse latency: %v", err)
	}
	conn, err := grpc.Dial(*pipeAddr, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("failed to dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewInferencePipeClient(conn)
	req := &pb.FeaturesBatchRequest{BatchSize:uint32(*batchSize)}
	policy := make([]float32, numBridgeActions)
	ctx := context.Background()
	for k, _ := range policy {
		policy[k] = 1.0 / numBridgeActions
	}
	for k := 0; k < *numClients; k++ {
		go func(k int) { for l := 0; ; l++ {
			b, err := client.GetFeaturesBatch(ctx, req);
			time.Sleep(sleepInterval)
			if err != nil { log.Fatal(err) }
			micro := make([]*pb.PredictionsMicroBatch, len(b.Micro))
			for i, _ := range micro {
				p := make([]*pb.Prediction, 1)
				outcome_chord := &pb.Chord{MicroTokenId:[]uint32{15}}
				p[0] = &pb.Prediction{
					ValueGeq:0, ValueGt:0, Policy:policy,
					ParOutcome:outcome_chord}
				micro[i] = &pb.PredictionsMicroBatch{Prediction:p}
			}
			resp := &pb.PredictionsBatch{UniqueId:b.UniqueId, Micro:micro}
			_, err = client.PutPredictionsBatch(ctx, resp)
			if err != nil { log.Print(err) }
		} }(k)
	}
	c := make(chan struct{})
	_ = <-c
}
