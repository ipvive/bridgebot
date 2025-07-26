package wikimedia

import (
	"os"
	"testing"
)

func TestGet(t *testing.T) {
	os.Chdir("x/wikimedia")
	mi := MultiIndex{}
	var wikis = []string{"testdata/tstwiki", "x/wikimedia/testdata/otherwiki"}
	mi.LoadWikisFromPaths(wikis)
	_, b1 := mi.Get("tst", "Hello", false)
	_, b2 := mi.Get("tsl", "Hello", false)
	_, b3 := mi.Get("other", "Hello", false)
	_, b4 := mi.Get("tst", "World", false)
	if b1 == nil || b2 != nil || b3 != nil || b4 == nil || string(b1) == string(b4) {
		t.Errorf("b1=%s b2=%s b3=%s b4=%s", b1, b2, b3, b4)
	}
}
