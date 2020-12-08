<?php
	
?>
<!DOCTYPE html>
<html>
<head>
	<title>PROJECT</title>
	<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
	<script type="text/javascript">
		function get_conv() {
			var ques = document.getElementById("nl_question").value;
			var str = ques;

    		if (str.length == 0) {
        		document.getElementById("txtHint").innerHTML = "";
        	return;
    		} else {
        		var xmlhttp = new XMLHttpRequest();
        		xmlhttp.onreadystatechange = function() {
            	if (this.readyState == 4 && this.status == 200) {
            		document.getElementById("conv").style.display = "block";
                	document.getElementById("cnn_cat").innerHTML = this.responseText;
            		}
        		};
        		xmlhttp.open("GET", "php/get_conv.php?ques=" + str, true);
        		xmlhttp.send();
    		}
		}
		function get_seq() {
			var ques = document.getElementById("nl_question").value;
			var str = ques;

    		if (str.length == 0) {
        		document.getElementById("txtHint").innerHTML = "";
        	return;
    		} else {
        		var xmlhttp = new XMLHttpRequest();
        		xmlhttp.onreadystatechange = function() {
            	if (this.readyState == 4 && this.status == 200) {
	            		document.getElementById("sequence").style.display = "block";
	                	document.getElementById("nmt_query").innerHTML = this.responseText;
            		}
        		};
        		xmlhttp.open("GET", "php/get_seq.php?ques=" + str, true);
        		xmlhttp.send();
    		}
		}
		function get_seq1() {
			var ques = document.getElementById("nl_question").value;
			var str = ques;

    		if (str.length == 0) {
        		document.getElementById("txtHint").innerHTML = "";
        	return;
    		} else {
        		var xmlhttp = new XMLHttpRequest();
        		xmlhttp.onreadystatechange = function() {
            	if (this.readyState == 4 && this.status == 200) {
            			console.log(this.responseText);
	            		return this.responseText;
            		}
        		};
        		xmlhttp.open("GET", "php/get_seq.php?ques=" + str, true);
        		xmlhttp.send();
    		}
		}
		function get_query() {
			var ques = document.getElementById("nl_question").value;
			var str = ques;

    		if (str.length == 0) {
        		document.getElementById("txtHint").innerHTML = "";
        	return;
    		} else {
        		var xmlhttp = new XMLHttpRequest();
        		xmlhttp.onreadystatechange = function() {
            	if (this.readyState == 4 && this.status == 200) {
            			console.log(this.responseText);
	            		return this.responseText;
            		}
        		};
        		xmlhttp.open("GET", "php/get_query.php?ques=" + str, true);
        		xmlhttp.send();
    		}
		}
	</script>
</head>
<body>
	<label><h3>ENTER YOUR QUESTION HERE:</h3></label><br>
	<div style="text-align: center;">
		<input size=150 type="text" name="nl_question" id="nl_question"><br>
	</div>
	<br>
	<div style="text-align: center;">
		<button type="button" class="btn btn-outline-primary" onclick="get_conv()">GET CONVOLUTION CATEGORY</button>&nbsp
		<button type="button" class="btn btn-outline-primary" onclick="get_seq()">GET SEQUENCE</button>&nbsp
		<button type="button" class="btn btn-outline-success" onclick="get_query()">CONVERT TO QUERY</button>&nbsp
	</div>
	<br><br>
	<div id="conv" style="display: none;">
		<div class="card">
		  <div class="card-header">
		    TYPE OF QUERY:
		  </div>
	  		<div class="card-body">
    			<h5 class="card-title" id="cnn_cat"></h5>
    			<a href="#" class="btn btn-primary">CHECK THE CNN CONVERSION</a>
  			</div>
		</div>	
	</div>
	<br><br>

	<div id="sequence" style="display: none;">
		<div class="card">
		  <div class="card-header">
		    Predicted translation:
		  </div>
	  		<div class="card-body">
    			<h5 class="card-title" id="nmt_query"></h5>
    			<a href="#" class="btn btn-primary">SEE ATTENTION GRAPH</a>
  			</div>
		</div>	
	</div>
	<br><br>

	<div id="query" style="display: none;">
		<div class="card">
		  <div class="card-header">
		    Predicted translation after Reinforcement Learning:
		  </div>
	  		<div class="card-body">
    			<h5 class="card-title" id="rl_query"></h5>
    			<!-- <a href="#" class="btn btn-primary">Go somewhere</a> -->
  			</div>
		</div>	
	</div>
	<div >
		 
	</div>
	<div id="attention_img" style="display: none;">
		
	</div>
</body>
</html>
<?php
	if(isset($_POST['nl_question'])){
		// echo $_POST['nl_question'];
		$pass = $_POST['nl_question'];
		$cmd = "python C:\Kshitij\Dev\Jupyter\NMT_TRIAL_1.py"." ".$pass;
		$cmd1 = escapeshellcmd($cmd);
		$message = shell_exec($cmd1);
		print_r($message);
	}
?>