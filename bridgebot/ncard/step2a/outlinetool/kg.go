package main

type KG interface {
	// ...
}

type LocalPredicate string
type LocalEntity string

type LocalRelation struct {
	Subject LocalEntity
	Predicate LocalPredicate
	Object LocalEntity
}

type LocalKG struct {
	Entities []LocalEntity
	Relations []LocalRelation
}

type LocalKGFilter func(*LocalKG) *LocalKG

func Walk(src *KG, dest *KG, filter LocalKGFilter) {
	// ...
	// issue: we need multiple Walk impls, and don't always to bind at call site.
}
