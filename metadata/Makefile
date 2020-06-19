
all: 2-classes.yaml 3-classes.yaml 7-classes.yaml
clean:
	rm 2-classes.yaml 3-classes.yaml 7-classes.yaml

2-classes.yaml: reformat-metadata.py
	./reformat-metadata.py old/2-classes/*/*.data > 2-classes.yaml

3-classes.yaml: reformat-metadata.py
	./reformat-metadata.py old/3-classes/*/*.data > 3-classes.yaml

7-classes.yaml: reformat-metadata.py
	./reformat-metadata.py old/7-classes/*/*.data > 7-classes.yaml

