mkdir -p $2
cat $1 | grep "Val Error" | cut -c 27- > $2/val_$1
cat $1 | grep "Train Error" | cut -c 29- > $2/train_$1