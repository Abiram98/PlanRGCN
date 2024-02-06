dispather () {
   until [ -f $1 ]
   do 
	   sleep 7200
   done
   bash $1
}

experiment_funct () {
   (cd ... && timeout -s 2 7200 python3 -m load_balance.main_balancer conf.conf)
   touch $1
   sleep 30
   (cd ... && timeout -s 2 7200 python3 -m load_balance.main_balancer conf.conf)

}