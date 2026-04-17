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
            title = entry.get("title", "No title available")
            summary = entry.get("summary", "No summary available")
            link = entry.get("link", "#")

            articles.append({
                "title": title,
                "summary": summary,
                "link": link
            })

    return articles[:6]
