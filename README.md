# fuzzy-k-means

This is a Fuzzy-k-Means implementation according to [Ross T. (2010), Fuzzy Logic with Engineering Applications, p.152-153](http://www.amazon.de/Fuzzy-Logic-Engineering-Applications-Timothy/dp/047074376X) written in Scala with a `scikit-learn`-like API.

## Examples
To run the examples I recommend to use the Docker image `shokuninsan/fuzzylab` that I have prepared just for you. So make sure you have [Docker installed](https://docs.docker.com/engine/installation/) on your machine.

Run a Docker container from within the root of this project with `docker run -i -t -p 0.0.0.0:8888:8888 -v $(pwd):/home/pylab shokuninsan/fuzzylab:0.1.0`. On success you get a shell within the container (you will notice that your command prompt has changed, e.g. `root@cc85df33e59c:/#`).

Within the docker container change into the mounted volume `cd home/pylab/`.

From there you can run the IPython notebook: `ipython notebook --ip=0.0.0.0 --port=8888`.

To work with the notebook you need to access the webserver via your browser on your host system. To obtain the appropriate IP address of your container on Mac OS X, you need to execute `docker-machine ip default`. Now you can navigate your browser to `<your-ip>:8888/tree/examples/visualization`, select `iris.ipynb` and in the menu bar click _Cell_ > _Run All_.

If you want to trigger some knobs on `KMeans` (e.g. change `fuzziness` or `numClusters` parameters), just open the `iris.amm` script within the IPython notebook in your browser, change it as you desire and re-run the related cells in `iris.ipynb`.

Note: you can reconnect to a running docker container like this

	$ docker ps
	CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS                              NAMES
	cc85df33e59c        plotlab             "/bin/bash"         8 hours ago         Up 8 hours          0.0.0.0:8000-8001->8000-8001/tcp   dreamy_joliot

	$ docker exec -i -t cc85df33e59c bash
