lb_wikidata_qpp2 (){
	    config_path=/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_WD_load_balance.ini
	        CPUS="10"
		    dbpath=/data/abiram/wdbench/virtuoso_dabase
		        imp_path=/data/abiram/wdbench/import
			    docker run --rm -v $dbpath:/database ubuntu bash -c "rm /database/virtuoso.trx"
			        docker run -m 64g --rm --name wdbench_virt -d --tty --env DBA_PASSWORD=dba --env DAV_PASSWORD=dba --publish 1112:1111 --publish 8891:8890 -v $dbpath:/database -v imp_path:/import -v $config_path:/database/virtuoso.ini --cpus=$CPUS openlink/virtuoso-opensource-7:latest
}

#ARG1 filepath  when the database is reaady to be restarted
#ARG2 filepath to when the 
db_restarter (){
   until [ -f $1 ]
   do
	   sleep 600 # checks every 10 min
   done
   docker rm wdbench_virt
   lb_wikidata_qpp2()
   sleep 180
   touch $2
}

