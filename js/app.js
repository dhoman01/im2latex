(function(){
  angular.module('im2latex', ['ngRoute', 'ngMaterial'])
  .config(function($routeProvider, $mdThemingProvider){
    $routeProvider
    .when('/', {
      templateUrl: 'views/home.html'
    })
    .otherwise({
      redirectTo: '/'
    })

    var shingoRedMap = $mdThemingProvider.extendPalette('red',{
      '500': '#640921'
    });
    var shingoBlueMap = $mdThemingProvider.extendPalette('blue',{
      '500': '#003768'
    })

    $mdThemingProvider.definePalette('shingoBlue', shingoBlueMap);
    $mdThemingProvider.definePalette('shingoRed', shingoRedMap);
    $mdThemingProvider.theme('default')
    .primaryPalette('shingoRed', {'default':'500'})
    .accentPalette('shingoBlue', {'default': '500'});
  })
})();
