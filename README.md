# fuzzy-k-means

## Prerequisites
Make sure you have Docker and Java 8 installed on your machine.

## Run examples
1. `cd examples`
2. `java -jar ../deliverables/fuzzy-k-means-assembly-0.1.0.jar`
3. `docker run -i -t -p 0.0.0.0:8000:8000 -p 0.0.0.0:8001:8001 -v $(pwd):/home/pylab narenarya/plotlab`

Within the docker environment `cd home/pylab/ && ipython notebook --ip=0.0.0.0 --port=8000`. On Mac OS X you need to execute `docker-machine ip default` to get the ip address. Then navigate your browser to <ip>:8000.
