dispather () {
   until [ -f $1 ]
   do 
	   sleep 7200
   done
   bash $1
}

