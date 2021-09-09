#!/bin/bash

for node in sr610 sr612 sr613
do
        echo $node
        scp -r /root/ht/ML/wd-code/wd_sigopt $node:/root/ht/ML/wd-code > /dev/null
done

