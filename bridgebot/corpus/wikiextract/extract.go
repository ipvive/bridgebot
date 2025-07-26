package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/xml"
	"fmt"
	"log"
	"math/rand"
	"net/url"
	"os"
	"strings"

	"ipvive/bridgebot/x/wikimedia"
)

type Page struct {
	Id string `xml:"id"`
	Title string `xml:"title"`
	Revision struct {
		Text string `xml:"text"`
		Id string `xml:"id"`

	} `xml:"revision"`
}

const (
	corpus_dir = "data/corpus"
	categories_file = "bridge_categories"
	include_probability = 1e-3
)

func main() {
	var categories []string
	f, err := os.Open(categories_file)
	if err != nil {
		log.Print(err)
		os.Exit(1)
	}
	cs := bufio.NewScanner(f)
	for cs.Scan() {
		categories = append(categories, cs.Text())
	}
	f.Close()

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Split(wikimedia.ScanPages)
	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 64*1024*1024)
	for scanner.Scan() {
		var p Page
		err := xml.Unmarshal([]byte(scanner.Text()), &p)
		if err == nil {
			include_page := false
			for _, c := range(categories) {
				if strings.Index(p.Revision.Text, c) > 0 {
					include_page = true
					break
				}
			}
			if !include_page {
				if rand.Float32() > include_probability {
					continue
				}
			}


			url := fmt.Sprintf("wikiextract?title=%s&pageid=%s&revisionid=%s", url.QueryEscape(p.Title), p.Id, p.Revision.Id)
			urlid := fmt.Sprintf("%x", sha256.Sum256([]byte(url)))
			fn := fmt.Sprintf("%s/%s.txt", corpus_dir, urlid)
			f, err := os.Create(fn)
			if err != nil {
				log.Print(err)
				continue
			}
			_, err = f.WriteString(p.Revision.Text)
			if err != nil {
				log.Print(err)
				continue
			}
			err = f.Close()
			if err != nil {
				log.Print(err)
				continue
			}
			fmt.Printf("%s %s\n", url, urlid)
		} else {
			log.Print(err)
		}
	}
}
