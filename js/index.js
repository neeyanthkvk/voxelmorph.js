import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

"use strict";

var fs = require('fs');
var nifti = require('nifti-js');
var arrayBufferToBuffer = require('arraybuffer-to-buffer');
var fFile = null, sFile = null, buf1 = null, buf2 = null, ten1 = null, ten2 = null;

let model;

const MODEL_PATH = "../data/model.json"

async function loadFirst() {
    fFile = document.getElementById("first").files[0];
	var reader1 = new FileReader();
	reader1.onload = function(e) {
		var databuf1 = reader1.result;
		var file1 = nifti.parse(databuf1);
		var arr1 = file1.data;
		ten1 = tf.tensor(arr1).reshape([256,256,256]);
	}
	reader1.readAsArrayBuffer(fFile);
}


async function loadSecond() {
    sFile = document.getElementById("second").files[0];
	var reader2 = new FileReader();
	reader2.onload = function(e) {
		var databuf2 = reader2.result;
		var file2 = nifti.parse(databuf2);
		var arr2 = file2.data;
		ten2 = tf.tensor(arr2).reshape([256,256,256]);
	}
	reader2.readAsArrayBuffer(sFile);
}



async function Everything() {
	document.getElementById('first')
      .addEventListener('change', async () => {
        loadFirst();
    });
	document.getElementById('second')
      .addEventListener('change', async () => {
        loadSecond();
    });
	document.getElementById('button')
      .addEventListener('click', async () => {
        Evaluate();
    });	
}

async function Evaluate() {
	if(ten1 == null || ten2 == null) {
		console.log("One or More Files Not Loaded");
	}
	else {
		const jsonUpload = document.getElementById('json-upload');
		const weightsUpload = document.getElementById('weights-upload');
		const shardUpload = document.getElementById('shard-upload');
        const modell = tf.sequential();
        modell.add(tf.layers.conv3d({inputShape:[28,28,28,1],filters:32,kernelSize:3,activation:'relu',}));
        const model = await tf.loadModel(tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]));
		model.summary();
	}
}

Everything();
