compress() {
tar -czvf "$1".tar.gz "$2"
}

decompress(){
tar -xzvf "$1".tar.gz
}
