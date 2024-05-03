#!/bin/bash
export API_URL;

for i in ./src/environments/*.tmpl;do
    cat $i | envsubst > ${i%.*};
done;
