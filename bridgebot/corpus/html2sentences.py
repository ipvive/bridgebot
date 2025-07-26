import sys
import html2text
import nltk


def sentences(text):
    parts = nltk.sent_tokenize(text.replace("\n", " "))
    return parts

def markdown(html):
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True
    text_maker.ignore_emphasis = True

    text = text_maker.handle(html)
    return text

if __name__ == "__main__":
    for filename in sys.argv:
        with open(filename) as fp:
            lines = sentences(fp.read())
            print(lines.join("\n"))
