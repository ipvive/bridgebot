import scrapy
import glob
import pathlib

import pdb

class TournamentSpider(scrapy.Spider):
    name = 'tourneyspider'
    start_urls = [pathlib.Path(fn).absolute().as_uri()
                  for fn in glob.glob("data/*.html")]

    def parse(self, response):
        links = response.css("a::attr(href)").getall()
        for l in links:
            if l.startswith("fetchlin.php"):
                u = response.urljoin(f"https://www.bridgebase.com/{l}")
                yield scrapy.Request(u, callback=self.save)
    
    def save(self, response):
        pdb.set_trace()

