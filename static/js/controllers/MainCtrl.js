angular.module('MainCtrl', []).controller('MainController', function($scope, $rootScope, $timeout, $q, $log,$location) {

	$scope.tagline = 'Stock Price Prediction!';
	$scope.overView = function () {
	$location.path("/overview");
	}

		$scope.getStockPrice = function() {
			$log.info('Item changed to ' + JSON.stringify($scope.company));
			$rootScope.inputCompany = $scope.company;
			$location.path("/overview");
	      }
});
