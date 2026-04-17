import feedparser

def get_news():
    sources = [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "http://rss.cnn.com/rss/edition.rss"
    ]

    articles = []

    for url in sources:
        feed = feedparser.parse(url)

        for entry in feed.entries[:3]:
            articles.append({
                "title": entry.get("title", "No title"),
                "summary": entry.get("summary", "No summary available"),
                "link": entry.get("link", "#")
            })

    return articles[:6]
