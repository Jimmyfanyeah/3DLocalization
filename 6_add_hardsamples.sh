cp ../data_train/hardsamples/train/im*mat ../data_train/train
mv ../data_train/hardsamples/train/im*mat ../data_train/train_HS_b10_1
mv ../data_train/hardsamples/train/I*mat ../data_train/clean

cat ../data_train/hardsamples/train/label.txt >> ../data_train/train/label.txt
rm ../data_train/hardsamples/train/label.txt
