FROM ubuntu:16.04

# install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends\
    build-essential\
    git\
    cmake\
    ninja-build\
    python3-pip\
    python3-requests\
    python-dev\
    tzdata\
    sed\
    curl\
    wget\
    unzip\
    autoconf\
    libtool

RUN apt-get install -y mono-xbuild\
		mono-dmcs\
		mono-devel\
		 libmono-system-data-datasetextensions4.0-cil\
		libmono-system-web-extensions4.0-cil\
		libmono-system-management4.0-cil\
		libmono-system-xml-linq4.0-cil\
		libmono-windowsbase4.0-cil\
		libmono-system-io-compression4.0-cil\
		libmono-system-io-compression-filesystem4.0-cil\
		libmono-system-runtime4.0-cil\
		vim\
		unzip

RUN apt-get -y install ipython ipython-notebook


# you have to set your GitHub account here(name&password)
RUN git config --global user.name "Rong-Zhi"


# install protobuf via pip3
RUN pip3 install protobuf




