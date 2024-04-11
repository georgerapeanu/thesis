#!/bin/bash 

while true; do
  inotifywait -e move_self *.puml && java -jar plantuml-1.2024.4.jar *.puml -o targets
done
