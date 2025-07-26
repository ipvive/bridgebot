#include <arpa/inet.h>
#include <utility>
#include <string>

#include "re2/re2.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/io/table_options.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/file_system.h"

class IntegerKey : public tensorflow::StringPiece {
public:
  IntegerKey(int x) {
    data_[0] = x >> 24;
    data_[1] = x >> 16;
    data_[2] = x >> 8;
    data_[3] = x >> 0;
    // FIXME set(data_, 4);
  }
private:
  char data_[4];
};

void wikimediadumpsort(std::vector<std::string>* paths) {
  for (unsigned int i = 0; i < paths->size(); ++i) {
    for (unsigned int j = i + 1; j < paths->size(); ++j) {
      int iid = 0;
      int jid = 0;
      RE2::PartialMatch((*paths)[i], "xml-p(\\d+)p", &iid);
      RE2::PartialMatch((*paths)[j], "xml-p(\\d+)p", &jid);
      if (jid < iid) {
        std::swap((*paths)[i], (*paths)[j]);
      }
    }
  }
}

void process(tensorflow::Env *env, std::string wikiid) {
  tensorflow::FileSystem* infs;
  std::string input_glob = tensorflow::strings::Printf("%s/%s-pages-articles*.pageix", wikiid.c_str(), wikiid.c_str());
  std::vector<std::string> input_paths;
  std::string loc_sst_path = tensorflow::strings::Printf("%s/%s-pages-articles-loc.sst", wikiid.c_str(), wikiid.c_str());
  std::string title_sst_path = tensorflow::strings::Printf("%s/%s-pages-articles-title.sst", wikiid.c_str(), wikiid.c_str());
  std::string search_tsv_path = tensorflow::strings::Printf("%s/%s-pages-articles-search.tsv", wikiid.c_str(), wikiid.c_str());
  env->GetFileSystemForFile(wikiid, &infs);
  infs->GetMatchingPaths(input_glob, &input_paths);
  wikimediadumpsort(&input_paths);
  std::unique_ptr<tensorflow::WritableFile> f_loc, f_title, f_search;
  std::unique_ptr<tensorflow::ReadOnlyMemoryRegion> f_pageix;
  env->NewWritableFile(loc_sst_path, &f_loc);
  env->NewWritableFile(title_sst_path, &f_title);
  env->NewWritableFile(search_tsv_path, &f_search);
  auto options = tensorflow::table::Options();
  tensorflow::table::TableBuilder builder_loc(options, f_loc.get());
  tensorflow::table::TableBuilder builder_title(options, f_title.get());
  RE2 threeliner("(\\d+): *<page>[^<]*<title>([^<]*)</title>[^<]*<id>(\\d+)</id>\n");
  for (auto ifp : input_paths) {
    int offset;
    std::string title;
    int id;

    printf("Processing %s\n", ifp.c_str());
    env->NewReadOnlyMemoryRegionFromFile(ifp, &f_pageix);
    tensorflow::RegexpStringPiece input(static_cast<const char*>(f_pageix->data()), f_pageix->length());

    while (RE2::Consume(&input, threeliner, &offset, &title, &id)) {
      auto koffset = IntegerKey(offset);
      auto kid = IntegerKey(id);
      builder_loc.Add(kid, koffset);
      builder_title.Add(kid, title);
      f_search->Append(tensorflow::strings::Printf("%d\t", id));
      f_search->Append(title);
      f_search->Append("\n");
    }
  }
  builder_loc.Finish();
  builder_title.Finish();
}

int main(int argc, char** argv) {
  // tensorflow::port::InitMain(argv[0], &argc, &argv);
  auto env = tensorflow::Env::Default();
  for (int i = 1; i < argc; ++i) {
    process(env, argv[i]);
  }
}
