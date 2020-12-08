<?php
	if(isset($_REQUEST['ques'])){
		$question = $_REQUEST['ques'];
		$cmd = "python C:\\xampp\\htdocs\\BE_PROJ\\py\\NMT1.py"." ".$question;
		$cmd1 = escapeshellcmd($cmd);
		$message = shell_exec($cmd1);
		echo $message;
	}
?>