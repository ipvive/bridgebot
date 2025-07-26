package wikimedia

import (
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
)

var (
	reTitle = regexp.MustCompile(`    <title>(.*)</title>`)
	reLink = regexp.MustCompile(`\[\[([^]]*)\]\]`)
)

type packedEntry uint64

const (
	bitsFilenum = 7
	bitsOffset = 34
	bitsLength = 64 - bitsFilenum - bitsOffset
	
	shiftFilenum = 0
	shiftOffset = bitsFilenum
	shiftLength = shiftOffset + bitsOffset

	maskFilenum = ((1 << bitsFilenum) - 1)
	maskOffset = ((1 << bitsOffset) - 1)
	maskLength = ((1 << bitsLength) - 1)
)

func NewEntry(filenum, offset, length int) packedEntry {
	if (filenum & ^maskFilenum != 0) || (offset & ^maskOffset != 0) || (length & ^maskLength != 0) {
		log.Print("NewEntry: overflow")
		return packedEntry(0)
	}
        pFilenum := packedEntry(filenum) << shiftFilenum
	pOffset := packedEntry(offset) << shiftOffset
	pLength := packedEntry(length) << shiftLength
	return pFilenum | pOffset | pLength
}
func (e packedEntry) Offset() int { return int((e >> shiftOffset) & maskOffset) }
func (e packedEntry) Length() int { return int((e >> shiftLength) & maskLength) }
func (e packedEntry) Filenum() int { return int((e >> shiftFilenum) & maskFilenum) }
func (e packedEntry) Start() int { return e.Offset() }
func (e packedEntry) End() int { return e.Offset() + e.Length() }

type Index struct {
	paths []string
	data [][]byte // read-only. virtual.
	ix map[uint64]packedEntry
}

// LoadWikiFromPath is only safe on empty index.
func (i *Index) LoadWikiFromPath(dir string) {
	i.ix = make(map[uint64]packedEntry)
	// Find the files.
	filepath.Walk(dir, func(path string, f os.FileInfo, err error) error {
		if strings.Contains(path, ".xml") {
			fullPath, _ := filepath.Abs(path)
			i.paths = append(i.paths, fullPath)
		}
		return nil
	})

	// MMap the files.
	for _, fp := range i.paths {
		f, err := os.Open(fp)
		if err != nil {
			log.Fatalf("Open %s: %v", fp, err)
		}
		defer f.Close()
		fi, err := f.Stat()
		if err != nil {
			log.Fatalf("Stat %s: %v", fp, err)
		}
		size := fi.Size()
		d, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
		if err != nil {
			log.Fatalf("Failed to mmap %s: %v", fp, err)
		}
		i.data = append(i.data, d)
	}

	// Scan the data
	for ix, d := range i.data {
		i.ScanDumpFile(ix, d)
	}
}

func (i *Index) ScanDumpFile(fileIndex int, d []byte) {
	var offset int64
	for {
		advance, token, err := ScanPages(d[offset:], true)
		_ = err
		if token != nil {
			mt := reTitle.FindSubmatch(token)
			if mt == nil {
				log.Print("Failed to find title")
			} else {
				title := WikipediaFixTitle(string(mt[1]))
				//log.Printf("mt[1]=%q title=%q", string(mt[1]), title)
				h := UrlHash(title)
				e := NewEntry(fileIndex, int(offset), len(token))
				i.ix[h] = e
			}
		} else if advance == 0 {
			break
		}
		offset += int64(advance)
	}
}

type Location struct {
	Url    string
	Offset int
	Size   int
}

func (i *Index) Get(title string, locationOnly bool) (*Location, []byte) {
	h := UrlHash(string(title))
	e, ok := i.ix[h]
	if !ok { return nil, nil }
	loc := &Location{Url: i.paths[e.Filenum()], Offset: e.Offset(), Size: e.Length()}
	if locationOnly {
		return loc, nil
	} else {
		return loc, i.data[e.Filenum()][e.Start():e.End()]
	}
}

type MultiIndex map[string]*Index

func (mi MultiIndex) LoadWikisFromPaths(dirs []string) {
	for _, dir := range dirs {
		lc := filepath.Base(dir)
		pos := strings.Index(lc, "wiki")
		if pos <= 0 {
			log.Fatalf("%q may not be a wiki", dir)
		}
		lc = lc[:strings.Index(lc, "wiki")]
		mi[lc] = &Index{}
		mi[lc].LoadWikiFromPath(dir)
	}
}

func (mi MultiIndex) Get(lc string, title string, locationOnly bool) (*Location, []byte) {
	i, ok := mi[lc]
	if !ok { return nil, nil }
	return i.Get(title, locationOnly)
}
