import feedparser

def get_news():
    feeds = [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "http://rss.cnn.com/rss/edition.rss"
    ]

    articles = []

    for f in feeds:
        feed = feedparser.parse(f)

        for e in feed.entries[:3]:
            articles.append({
                "title": e.get("title", ""),
                "summary": e.get("summary", ""),
                "link": e.get("link", "#")
            })

    return articles[:6]
