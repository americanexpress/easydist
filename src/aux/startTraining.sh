# Copyright (c) 2019 American Express Travel Related Services Company, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

mkdir experiments > /dev/null
mkdir experiments/exp$1 >/dev/null
mkdir experiments/exp$1/outs_train > /dev/null
mkdir experiments/exp$1/errs_train > /dev/null
sh run.sh &> out.txt &
pssh -P -v -t 0 -e /home/ec2-user/experiments/exp$1/errs_train -o /home/ec2-user/experiments/exp$1/outs_train -h /home/ec2-user/aux/host_file.txt -x "-i /home/ec2-user/aux/easyDist.pem -o \"StrictHostKeyChecking no\" " 'sh run.sh' 

input="/home/ec2-user/aux/resources.txt"

spotId=$(tail -n+1 $input | head -n1)
IFS=' ' read -ra spotId <<< "$spotId"
instanceId=$(tail -n+2 $input | head -n1)
IFS=' ' read -ra instanceId <<< "$instanceId"
ips=$(tail -n+3 $input | head -n1)

i=$(( -1 ))
for ip in ${ips[@]}; do
echo $i
        if [ $i -eq -1 ]; then
        	echo 'PS'
	else
                scp -i "/home/ec2-user/aux/easyDist.pem" ec2-user@"$ip":m_"$i"_logs.zip /home/ec2-user/experiments/exp$1/.
                scp -i "/home/ec2-user/aux/easyDist.pem" ec2-user@"$ip":m_"$i"_models.zip /home/ec2-user/experiments/exp$1/.
        fi
        i=$(( $i+1 ))
done
#sh /home/ec2-user/aux/endTraining.sh
