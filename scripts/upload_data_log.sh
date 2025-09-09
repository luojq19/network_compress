#!/bin/bash

tar -cf logs.tar logs/
tar -cf data.tar data/
rclone copy ./logs.tar dropbox:network_compress/
rclone copy ./data.tar dropbox:network_compress/
rm logs.tar data.tar