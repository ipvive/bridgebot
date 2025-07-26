package wikimedia

import (
	"bytes"
	"bufio"
	"testing"
)

func TestScanMissingFinal(t *testing.T) {
	buf := new(bytes.Buffer)
	buf.WriteString("abc  </siteinfo>\n  <page>\n")
        buf.WriteString("   <title>AccessibleComputing</title>")
        buf.WriteString("\n  <page>\n")
	s := bufio.NewScanner(buf)
	s.Split(ScanPages)
	s.Scan()
	if s.Text() != "  <page>\n   <title>AccessibleComputing</title>" {
		t.Errorf("got %q", s.Text())
	}
	notok := s.Scan()
	if false != notok {
		t.Errorf("too many: %q", s.Text())
	}
}

func TestScanFinal(t *testing.T) {
	buf := new(bytes.Buffer)
	buf.WriteString("abc  </siteinfo>\n  <page>\n")
        buf.WriteString("some random text")
        buf.WriteString("\n</mediawiki>")
	s := bufio.NewScanner(buf)
	s.Split(ScanPages)
	s.Scan()
	if s.Text() != "  <page>\nsome random text" {
		t.Errorf("got %q", s.Text())
	}
	notok := s.Scan()
	if false != notok {
		t.Errorf("too many: %q", s.Text())
	}
}
