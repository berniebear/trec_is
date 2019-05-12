
for folderName in "BERT" "fasttextCrawl" "gloveAndFasttext" "hashTag" "skipThoughts"
do
    cd $folderName && bash get_embedding.sh
    cd ..
done
