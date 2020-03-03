for i in {1..10}
do
	echo bash starting seed$i 
	mkdir seed$i
	python main.py -seed=$i
	cp -r train seed$i/
done
