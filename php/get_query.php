<?php
	if(isset($_REQUEST['ques'])){
		$question = $_REQUEST['ques'];
		$cmd = "python C:\\xampp\\htdocs\\BE_PROJ\\py\\pred_RL.py"." ".$question;
		$cmd1 = escapeshellcmd($cmd);
		$message = shell_exec($cmd1);
		echo $message;
	}
		$question = "GIVE ME ALL THE NAMES";
		$cmd = "python C:\\xampp\\htdocs\\BE_PROJ\\py\\pred_RL.py"." ".$question;
		$cmd1 = escapeshellcmd($cmd);
		$message = shell_exec($cmd1);
		echo $message;
?>