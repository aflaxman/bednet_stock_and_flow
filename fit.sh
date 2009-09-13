#
# Below you will find options used by SGE to affect how
# your job is queued and run.
#
# Use the bash shell to interpret this job script
#$ -S /bin/bash
#
# Remove one # to send an e-mail to the address 
# specified in .sge_request when this job ends.
##$ -m e
#
# only submit this job to nodes
# that have at least 8GB of RAM free.
#$ -l mem_free=.9G


# to run this script for all countries, do the following
## for i in {0..50}; do qsub fit.sh $i; done

## Put the hostname, current directory, and start date
## into variables, then write them to standard output.
GSITSHOST=`/bin/hostname`
GSITSPWD=`/bin/pwd`
GSITSDATE=`/bin/date`
echo "**** JOB STARTED ON $GSITSHOST AT $GSITSDATE"
echo "**** JOB RUNNING IN $GSITSPWD"
##

cd /net/gs/vol1/home/abie/bednet_stock_and_flow/
echo calling bednets.py "$@"
/usr/local/epd_py25-4.3.0/bin/python bednets.py "$@"


## Put the current date into a variable and report it before we exit.
GSITSENDDATE=`/bin/date`
echo "**** JOB DONE, EXITING 0 AT $GSITSENDDATE"
##

## Exit with return code 0, otherwise the job would exit with the 
## return code of the failed ls command.
exit 0

