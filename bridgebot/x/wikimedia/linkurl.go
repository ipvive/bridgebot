package wikimedia

import (
	"bytes"
	"fmt"
	"regexp"
	"strings"
	"net/url"

	"github.com/dchest/siphash"
        "github.com/golang/glog"
)

var (
	re_lc = regexp.MustCompile(`^:?(\w\w):`)
	re_ws = regexp.MustCompile("[_ \t]+")
	re_wikilanguage = regexp.MustCompile("([a-z]+).wikipedia.org")
	re_wikixmltitle = regexp.MustCompile("titles=(.*)&rvprop=content")
	re_wikihtmltitle = regexp.MustCompile(".wikipedia.org/wiki/(.*)")
	url_hash_k0 uint64 = 67578678578
	url_hash_k1 uint64 = 89756644312
)

func WikipediaURLToLanguageCodeAndTitle(u string) (lc, title string) {
	if m := re_wikilanguage.FindStringSubmatch(u); m != nil {
		lc = m[1]
	}
	if m := re_wikixmltitle.FindStringSubmatch(u); m != nil {
		title = m[1]
	} else if m := re_wikihtmltitle.FindStringSubmatch(u); m != nil {
		title = m[1]
	} else {
		glog.Fatalf("Invalid wikipedia URL %q", u)
	}
	title, _ = url.PathUnescape(title)
	title = WikipediaFixTitle(title)
	return lc, title
}

func WikipediaLinkToCanonicalUrl(language_code, link string) string {
	if m := re_lc.FindStringSubmatch(link); m != nil {
		language_code = m[1]
		link = strings.TrimPrefix(link, m[0])
	}
	parts := strings.Split(link, "|")
	parts = strings.Split(parts[0], "#")
	title := WikipediaFixTitle(parts[0])
 	title = strings.Replace(title, " ", "_", -1)
	return fmt.Sprintf("https://%s.wikipedia.org/w/api.php?action=query&format=xml&prop=revisions&titles=%s&rvprop=content", language_code, url.PathEscape(title))
}

func WikipediaFixTitle(title string) string {
	title = strings.Trim(title, " \t")
	if title == "" {
		return ""
	}
	first := title[:1]
	upper := strings.ToUpper(first)
	title = strings.Replace(title, first, upper, 1)
	title = re_ws.ReplaceAllLiteralString(title, " ")
	return title
}

func CanonicalUrl(url string) string {
	lc, title := WikipediaURLToLanguageCodeAndTitle(url)
	return WikipediaLinkToCanonicalUrl(lc, title)
}

func UrlHash(url string) uint64 {
	b := bytes.NewBufferString(url)
	return siphash.Hash(url_hash_k0, url_hash_k1, b.Bytes())
}
