# pulse-aggregator
Aggregate events by pulse and convert wallclock detection times to time relative to pulse.

Initial implementation reads from, and writes to, NeXus files. Working towards having it work from and to Kafka stream.

Requires Python 3.6+.
To install required Python packages:
```bash
pip install -r requirements.txt
```

Please then install the pre-commit hook to run [Black](https://github.com/python/black) formatter:
```bash
pre-commit install
```
