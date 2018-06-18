# Twitter Squares

Arranges profile images from a twitter search in a [square t-sne](https://github.com/prabodhhere/tsne-grid/) using the [Jonker-Volgenan](https://blog.sourced.tech/post/lapjv/) algorithm. Features are generated from VGG16 (without fc layers on top).

## `#bitcoin`
![](docs/examples/bitcoin.jpg)

## `#cardi`
![](docs/examples/cardi.jpg)

## [More examples...](EXAMPLES.md)

### Usage

1] Install the dependencies `pip install -r requirements.txt`.

2] Create a file `credentials.ini` and fill out your information:

```
[TwitterCredentials]
consumer_key = ...
consumer_secret = ...
access_token = ...
access_token_secret = ...
```

3] Then to pull 144 (12x12) from the search `obama`

     python scrape.py obama 144
     python render.py obama 144
