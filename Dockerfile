FROM narenarya/plotlab:latest

USER root

# add webupd8 repository
RUN \
    echo "===> add webupd8 repository..."  && \
    echo "deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main" | tee /etc/apt/sources.list.d/webupd8team-java.list  && \
    echo "deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main" | tee -a /etc/apt/sources.list.d/webupd8team-java.list  && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys EEA14886  && \
    apt-get update  && \
    \
    \
    echo "===> install Java"  && \
    echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections  && \
    echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections  && \
    DEBIAN_FRONTEND=noninteractive  apt-get install -y --force-yes oracle-java8-installer oracle-java8-set-default  && \
    \
    \
    echo "===> clean up..."  && \
    rm -rf /var/cache/oracle-jdk8-installer  && \
    apt-get clean  && \
    rm -rf /var/lib/apt/lists/*


# Install sbt
RUN wget http://dl.bintray.com/sbt/debian/sbt-0.13.11.deb && \
  dpkg -i sbt-0.13.11.deb
RUN apt-get update && apt-get install sbt

# Install ammonite
RUN apt-get update && apt-get install -y curl
RUN curl -L -o amm https://git.io/vafIQ && chmod +x amm && mv amm /usr/local/bin

# Publish fuzzy-k-means into local .ivy2 repo
RUN apt-get update && apt-get install -y git
RUN git clone --branch 0.1.0-SNAPSHOT https://github.com/ShokuninSan/fuzzy-k-means.git
RUN cd fuzzy-k-means && sbt publishLocal

# Install plotting module for visualization
RUN cd fuzzy-k-means/examples/visualization && python setup.py install