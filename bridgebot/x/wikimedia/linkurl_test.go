package wikimedia

import (
	"testing"
)

func TestWikipediaLinkToUrl(t *testing.T) {
	cases := []struct {
		lang string
		link string
		expectedUrl string
	}{
		{
			lang: "en",
			link: "Bitcoin",
			expectedUrl: "https://en.wikipedia.org/w/api.php?action=query&format=xml&prop=revisions&titles=Bitcoin&rvprop=content",
		},
		{
			lang: "en",
			link: "ru:Bitcoin",
			expectedUrl: "https://ru.wikipedia.org/w/api.php?action=query&format=xml&prop=revisions&titles=Bitcoin&rvprop=content",
		},
		{
			lang: "es",
			link: "Bitcoin",
			expectedUrl: "https://es.wikipedia.org/w/api.php?action=query&format=xml&prop=revisions&titles=Bitcoin&rvprop=content",
		},
		{
			lang: "en",
			link: "Bitcoin#section",
			expectedUrl: "https://en.wikipedia.org/w/api.php?action=query&format=xml&prop=revisions&titles=Bitcoin&rvprop=content",
		},
		{
			lang: "en",
			link: "Bitcoin|Notcoin|other",
			expectedUrl: "https://en.wikipedia.org/w/api.php?action=query&format=xml&prop=revisions&titles=Bitcoin&rvprop=content",
		},
		{
			lang: "en",
			link: "\tThe Wall\tStreet\t \tJournal ",
			expectedUrl: "https://en.wikipedia.org/w/api.php?action=query&format=xml&prop=revisions&titles=The_Wall_Street_Journal&rvprop=content",
		},
	}
        for _, c := range cases {
		url := WikipediaLinkToCanonicalUrl(c.lang, c.link)
		if c.expectedUrl != url {
			t.Errorf("Expected %q but got %q", c.expectedUrl, url)
		}
	}
}

func TestWikipediaURLToLanguageCodeAndTitle(t *testing.T) {
	url := "https://en.wikipedia.org/w/api.php?action=query&format=xml&prop=revisions&titles=Geometrization_conjecture&rvprop=content"
	lc, title := WikipediaURLToLanguageCodeAndTitle(url)
	if lc != "en" || title != "Geometrization conjecture" {
		t.Errorf("Want en, Geometrization conjecture, got %q, %q", lc, title)
	}
}

func TestCanonicalUrl(t *testing.T) {
	cases := []struct{
		Canonical string
		Variants []string
	}{
		{
			Canonical: "https://en.wikipedia.org/w/api.php?action=query&format=xml&prop=revisions&titles=Vexatious_litigation&rvprop=content",
			Variants: []string{
				"https://en.wikipedia.org/wiki/Vexatious_litigation",
				"https://en.wikipedia.org/wiki/Vexatious litigation",
				"https://en.wikipedia.org/wiki/vexatious_litigation",
			},
		},
		{
			Canonical: "https://ko.wikipedia.org/w/api.php?action=query&format=xml&prop=revisions&titles=%EC%82%AC%EC%A0%81%EC%9E%90%EC%B9%98%EC%9D%98_%EC%9B%90%EC%B9%99&rvprop=content",
			Variants: []string{
				"https://ko.wikipedia.org/wiki/%EC%82%AC%EC%A0%81%EC%9E%90%EC%B9%98%EC%9D%98_%EC%9B%90%EC%B9%99",
			},
		},
	}
	
	for _, c := range cases {
		for _, v := range c.Variants {
			actual := CanonicalUrl(v)
			if actual != c.Canonical {
				t.Errorf("%q: Want %q, Got %q", v, c.Canonical, actual)
			}
		}
	}
}

func TestUrlHash(t *testing.T) {
	actual := UrlHash("Quantified Happier Humanity")
	if 0x5f5081291cca7584 != actual {
		t.Errorf("Expected 0x5f5081291cca7584 but got 0x%x", actual)
	}
}
