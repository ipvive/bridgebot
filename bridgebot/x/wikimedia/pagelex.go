package wikimedia

import (
	"bytes"
	"log"
)

var (
	pageSep = []byte("\n  <page>\n")
	finalSep = []byte("\n</mediawiki>")
)

// ScanPages is a split function for a bufio.Scanner that returns each
// page in a wikimedia xml dump.
func ScanPages(data []byte, atEOF bool) (advance int, token []byte, err error) {
	// log.Printf("scan %d %v...", len(data), string(data[0:100]))
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}
	if i := bytes.Index(data, pageSep); i >= 0 {
		if data[0] != ' ' {
			// Skip the header information.
			return i + 1, nil, nil
		} else {
			// The leading newline is neither saved nor tokenized.
			return i + 1, data[0:i], nil
		}
	}
	if atEOF {
		i := bytes.Index(data, finalSep)
		if i > 0 {
			return len(data), data[0:i], nil
		} else {
			return 0, nil, nil
		}
	}
	// Done.
	return 0, nil, nil
}

func PageFeatures (pageurl string, page []byte) (title []byte, hrefs [][]byte) {
	mt := reTitle.FindSubmatch(page)
	if mt != nil {
		title = mt[1]
		lc, _ := WikipediaURLToLanguageCodeAndTitle(pageurl)
		pageurl2 := WikipediaLinkToCanonicalUrl(lc, string(title))
		if pageurl != pageurl2 {
			_ = log.Printf
//			log.Printf("inconsistent url: %q != %q", pageurl, pageurl2)
		}
		for _, l := range reLink.FindAll(page, -1) {
			refurl := WikipediaLinkToCanonicalUrl(lc, string(l[2:len(l)-2]))
			hrefs = append(hrefs, []byte(refurl))
		}
	}
	return
}
