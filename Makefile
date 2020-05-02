


TAR = tar
TARFLAGS = -cvf
EXCLUDE = --exclude
CHDIR = -C

TARNAME = sentimentAnalysis.tar
TARSRCS = app assets dev README.md
TAREXCLUDE = $(EXCLUDE) "app/.idea" $(EXCLUDE) "dev/__pycache__" $(EXCLUDE) "app/SATester.java"


tar:
	$(TAR) $(TARFLAGS) $(TARNAME) $(TAREXCLUDE) $(TARSRCS)
