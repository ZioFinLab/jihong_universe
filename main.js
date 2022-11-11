(function ($) {
  
  var debug = $.proxy(window.console, 'debug');
  
  $(document).ready(function () {
    
    var $form = $('form')
    
    $('input[name=attachment]')
        .css('width', $('button.upload-file').outerWidth()) /* cover the same area! */
        .on('mouseover', function (ev) {
            //debug('About to select a file')
            return true;
        })
        .on('mouseout', function (ev) {
            //debug('Nevermind')
            return true;
        })
        .on('change', function (ev) {
            var fileobj = $(this).prop('files').item(0)
            debug('An attachment was selected:', fileobj)
            $('.selected-file .name').text(fileobj.name)
            $('.selected-file').show()
        });
    
    $('button.upload-file')
        .on('click', function () {
            warn('I should have been intercepted by input:file !')
            return false;
        });
    
    $('button.link-file')
        .on('click', function () {
            debug('button.link-file clicked')
            // Todo Let user provide a link (instead of an upload)
            return false;
        });
    
    $('.selected-file > .upload-file')
        .on('click', function () {
            var opts = {
                type: 'post',
                dataType: 'json', /* force response content-type */
                iframe: true, /* force transport over iframe */
                data: $form.serializeArray(),
                files: $form.find('input:file'),
            };
            debug('Starting upload ..')
            $.ajax($form.prop('action'), opts)
                .done(function (data) {
                    debug('Finished, response is:', data)
                })
        })

  });
  
})(jQuery);
