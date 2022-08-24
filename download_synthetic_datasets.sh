# Copyright 2022 Juan L. Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Download and unzip the synthetic datasets in synthetic_experiments/

URLS=(
    "https://polybox.ethz.ch/index.php/s/AnZzOVZ7tv32V9o/download"
    "https://polybox.ethz.ch/index.php/s/8xz0zL6YTnl52sM/download"
    "https://polybox.ethz.ch/index.php/s/b4xpDUanh5S4Hdw/download"
    "https://polybox.ethz.ch/index.php/s/gL09gDoxjzYqNuG/download"
    )

SAVEPATH="synthetic_experiments/"


# ---------------------------------
# Check that commands are installed

if ! command -v wget &> /dev/null
then
    echo 'ERROR: "wget" package needs to be installed to dowbload the datasets'
    exit
fi

if ! command -v unzip &> /dev/null
then
    echo 'ERROR: "unzip" package needs to be installed to unpack the datasets'
    exit
fi


# ---------------------------------
# Download datasets

echo "Downloading synthetic datasets"
echo
for URL in ${URLS[@]}
do
    wget --content-disposition -P $SAVEPATH $URL
    if [ $? -ne 0 ]
    then
        echo "ERROR (see above): Could not download dataset from url: "$URL
        exit 1
    fi
       
done


# ---------------------------------
# Unzip datasets

echo "Downloaded datasets. Unpacking..."
echo
for FILE in $SAVEPATH/dataset_*.zip
do
    unzip $FILE -d $SAVEPATH/synthetic_experiments/
    if [ $? -ne 0 ]
    then
        echo "ERROR (see above): Could not unzip dataset from file: "$FILE
        exit 1
    fi
done

echo
echo "Succesfully downloaded and unpacked datasets :)"
