<?php 

header('Content-Type: application/json; charset=utf-8');

$result = array(
    'params' => $_POST,
    'files' => $_FILES,
);

echo json_encode($result);

?>
