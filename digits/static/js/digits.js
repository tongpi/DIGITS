// Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.

function errorAlert(response) {
    var title, msg;
    if (response.status == 0) {
        title = '{{_("An error occurred!")}}';
        msg = '<p class="text-danger">{{_("The server may be down."}}</p>';
    } else {
        title = response.status + ': ' + response.statusText;
        msg = response.responseText ? response.responseText : response.data;
    }
    bootbox.alert({
        title: title,
        message: msg,
    });
}
