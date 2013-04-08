declare -a myarray
while read p; do
          find . -name $p -print >> git-ignore-these.sh
done < .gitignore

sed -i 's/.*/git rm --cached &/' git-ignore-these.sh
sed -i '1s/.*/#!\/bin\/bash \
              &/' git-ignore-these.sh
chmod +x git-ignore-these.sh

#sed 's/\\/\\\\/' git-ignore-these.sh >>  git-ignore-these.sh
