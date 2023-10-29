import app

def test_tweet_clean():
    tweet = "This is a test" 
    assert app.tweet_clean(tweet) == "this is a test"

def test_tweet_predict():
    tweet = "This is a test"
    assert type(app.tweet_predict(tweet)) == float

def test_tweet_sentiment_positif():
    pred = 0.8
    assert app.tweet_sentiment(pred) == "Positif"
    
def test_tweet_sentiment_negatif():
    pred = 0.3
    assert app.tweet_sentiment(pred) == "Negatif"
