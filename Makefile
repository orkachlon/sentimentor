

TAR = tar
TARFLAGS = -cvf

TARNAME = sentimentAnalysis.tar
TARSRCS = app/*.java app/*.pde app/*.bat assets/vectorizers dev README.md
TAREXCLUDE = --exclude "dev/__pycache__" --exclude "dev/test" --exclude "app/SATester.java"


tar:
	$(TAR) $(TARFLAGS) $(TARNAME) $(TAREXCLUDE) $(TARSRCS)
