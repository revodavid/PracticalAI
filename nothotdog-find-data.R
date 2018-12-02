## Find images to use for the Not Hotdog with Custom Vision application

## DO NOT RUN THIS SCRIPT
## This script generates the supplied hotdog-good.txt and nothotdog-good.txt files
## It samples from a collection of ImageNet URLs in various categories
## and eliminates URLs that fail and other inappropriate images

library(tools)
library(httr)

## these lists were downloaded from ImageNet 2011 http://image-net.org/explore_popular.php
## for the "hot dog", "frankfurter bun",  categories
dogs <- scan("hotdogs.txt",what=character())
franks <- scan("frankfurtbuns.txt", what=character())
burgers <- scan("hamburger.txt", what=character())
tacos <- scan("tacos.txt", what=character())

## so we can reproduce this later
set.seed(3302018)

## We won't need many images to build a Not Hotdog classifier, so let's grab
## 100 images. We'll actually lose about 30% to bad URLs further down the line
Nimages <- 100
hotdogs <- sample(c(dogs,franks),Nimages)

## We'll also grab images of things that might be easily mistaken for hotdogs:
## burgers, tacos
nothotdogs <- sample(c(burgers,tacos), Nimages)

## Not all of the URLs are valid. URLs may not be accessible, and Flickr may return
## a "This photo is not available" image
## These indexes were calculated using code in the next chunk and visual inspection.
## They're only valid if you use the same seed, Nimages, and supplied .txt files of URLs
bad.dogs <- c(3,4,10,14,18,21,25,26,28,32,33,34,38,43,47,51,56,68,71,73,75,76,79,80,81,82,85,89,91,94,95,96,
              31, #woman eating icecream
              52, #redirected URL
              62, #puppy in a hotdog bun
              88, #loaves of bread,
              89 #invalid URL
)

bad.notdogs <- c(8,9,12,26,27,31,34,50,52,56,61,69,71,72,83,86,95,97,98,
                 54, # bad extension
                 93  #corrupt file
)

hotdogs.good <- hotdogs[-bad.dogs]
nothotdogs.good <- nothotdogs[-bad.notdogs]

write(hotdogs.good, "hotdogs-good.txt")
write(nothotdogs.good, "nothotdogs-good.txt")

## This code was run interactively. It's included here so you can see
## how we downloaded the images for review, and created the bad.dogs and bad.notdogs
## objects above
if(FALSE) {
 valid.dogs <- rep(TRUE, length(hotdogs))
 valid.dogs[bad.dogs] <- FALSE
 for (i in seq(along=hotdogs)) {
  u <- hotdogs[i]
  destfile <- file.path("hotdogs", paste0(i,"-",basename(u)))
  if(valid.dogs[i] && !file.exists(destfile))
   try(download.file(u, destfile , mode="wb", method="auto"))
  # flag failed downloads and tiny files (flickr error images are 2051 bytes)
  valid.dogs[i] <- file.exists(destfile) && (file.size(destfile)>9000)
 }
 # print index of bad URLs
 cat((1:Nimages)[!valid.dogs],sep=",")
 
 valid.notdogs <- rep(TRUE, length(nothotdogs))
 valid.notdogs[bad.notdogs] <- FALSE
 for (i in seq(along=nothotdogs)) {
  u <- nothotdogs[i]
  destfile <- file.path("nothotdogs", paste0(i,"-",basename(u)))
  if(valid.notdogs[i] && !file.exists(destfile))
   try(download.file(u, destfile , mode="wb", method="auto"))
  # flag failed downloads and tiny files (flickr error images are 2051 bytes)
  valid.notdogs[i] <- file.exists(destfile) && (file.size(destfile)>9000)
 }
 # print index of bad URLs
 cat((1:Nimages)[!valid.notdogs],sep=",")

}