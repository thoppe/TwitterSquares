# Twitter Squares

Arranges profile images from a twitter search in a [square t-sne](https://github.com/prabodhhere/tsne-grid/) using the [Jonker-Volgenan](https://blog.sourced.tech/post/lapjv/) algorithm. Features are generated from VGG16 (without fc layers on top).

## `#bitcoin`
![](docs/examples/bitcoin.jpg)

## `#cardi`
![](docs/examples/cardi.jpg)

## [More examples...](EXAMPLES.md)

### Usage

Create a file `credentials.ini` and fill out your information:

```
[TwitterCredentials]
consumer_key = ...
consumer_secret = ...
access_token = ...
access_token_secret = ...
```

Then to pull hits from the search `obama`

     python pull_from_search.py obama
     python render.py obama