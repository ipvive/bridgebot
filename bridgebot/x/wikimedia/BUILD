load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

filegroup(
    name = "testdata",
    srcs = glob(["testdata/**"]),
)

#cc_binary(
#    name = "pageix2tables",
#    srcs = ["pageix2tables.cc"],
#    deps = [
#        "@org_tensorflow//tensorflow/core:lib",
#    ],
#)

go_test(
    name = "go_default_test",
    srcs = [
        "hashtable_test.go",
        "linkurl_test.go",
        "pagelex_test.go",
    ],
    data = [":testdata"],
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "hashtable.go",
        "linkurl.go",
        "pagelex.go",
    ],
    importpath = "ipvive/x/wikimedia",
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_dchest_siphash//:go_default_library",
        "@com_github_golang_glog//:go_default_library",
    ],
)
